import { Datagrid, List, TextField } from 'react-admin';

const defaultItems = [
    { 'source': 'id', label: 'ID' },
    { 'source': 'anonymised_vin', label: 'Anonymised VIN' },
    { 'source': 'i_incident_date', label: 'Incident Date' },
    { 'source': 'i_original_ccc_description', label: 'Original CCC Description' },
    { 'source': 'i_cpsc_description', label: 'CPSC Description' },
    { 'source': 'i_p_css_description', label: 'P-CSS Description' },
    { 'source': 'i_css_description', label: 'CSS Description' },
    { 'source': 'ic_customer_verbatim', label: 'Customer Verbatim' },
    { 'source': 'ic_technical_verbatim', label: 'Technical Verbatim' },
]

export const WarrantyList = ({ items = defaultItems, ...params }) => (
    <List {...params}>
        <Datagrid>
            {items.map(item => <TextField key={item.source} source={item.source} label={item?.label || item.source} />)}
        </Datagrid>
    </List>
)
