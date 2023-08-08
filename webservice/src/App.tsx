
import { Admin, Resource, ShowGuesser, fetchUtils, Show, ShowBase } from 'react-admin'
import postgrestRestProvider, 
     { IDataProviderConfig, 
       defaultPrimaryKeys, 
       defaultSchema } from '@raphiniert/ra-data-postgrest'
import { WarrantyList } from './warranty/WarrantyList'
import { DiagnosticList } from './diagnostic/DiagnosticList'
import { ConsultationList } from './consultation/ConsultationList'
import { ConsultationShow } from './consultation/ConsultationShow'

// Theme from https://demos.adminmart.com/free/react/modernize-react-lite/landingpage/index.html
import { baselightTheme } from './theme/DefaultColors'
import CustomLayout from './layouts/CustomLayout'
import { API_URI } from './config'


const config: IDataProviderConfig = {
    apiUrl: API_URI,
    httpClient: fetchUtils.fetchJson,
    defaultListOp: 'eq',
    primaryKeys: defaultPrimaryKeys,
    schema: defaultSchema
}

export const App = () => (
    <Admin dataProvider={postgrestRestProvider(config)} theme={baselightTheme} layout={CustomLayout}>
        <Resource name="consultation" list={ConsultationList} show={<ShowBase><ConsultationShow /></ShowBase>} options={{
            label: 'Consultations'
        }}/>
        <Resource name="vehicles" list={DiagnosticList} show={ShowGuesser} options={{
            label: 'Diagnostic Data'
        }} />
    </Admin>
)
